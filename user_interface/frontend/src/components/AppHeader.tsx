import { Link } from 'react-router-dom'
import { ArrowLeft, Sprout } from 'lucide-react'
import { Button } from '@mui/material'
import type { ReactNode } from 'react'

interface AppHeaderProps {
  action?: ReactNode
  backTo?: string
}

export function AppHeader({ action, backTo }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__leading">
          <Link className="app-brand" to="/" aria-label="FarmAI home">
            <span className="app-brand__mark">
              <Sprout size={20} aria-hidden="true" />
            </span>
            FarmAI
          </Link>
          {backTo && (
            <Button
              component={Link}
              to={backTo}
              color="inherit"
              startIcon={<ArrowLeft size={18} />}
            >
              Back
            </Button>
          )}
        </div>
        {action}
      </div>
    </header>
  )
}
